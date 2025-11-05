# HTTP API Reference

ULEI provides a FastAPI-based HTTP server for online shadow evaluation of production traffic. This enables continuous monitoring and evaluation of your LLM/RAG systems in real-time.

## Starting the Server

### Basic Usage
```bash
# Start with evaluation suite configuration
ulei server --suite-config evaluation.yaml --port 8080

# Start with server configuration file
ulei server --config server_config.yaml --port 8080
```

### Development Mode
```bash
# Enable auto-reload on code changes
ulei server --suite-config evaluation.yaml --reload --verbose

# Set custom log level
ulei server --suite-config evaluation.yaml --log-level DEBUG
```

### Production Mode
```bash
# Multiple workers for high throughput
ulei server --config evaluation.yaml --host 0.0.0.0 --port 8000 --workers 4

# With access logging
ulei server --config evaluation.yaml --access-log
```

## API Endpoints

Base URL: `http://localhost:8080` (configurable)

All endpoints follow the OpenAPI specification at `specs/001-unified-eval-interface/contracts/http-api.yaml`.

### 1. Health Check

Check server health and processing status.

**Endpoint:** `GET /v1/health`

**Example:**
```bash
curl http://localhost:8080/v1/health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-11-05T10:00:00Z",
  "queue_size": 5,
  "processing_active": true
}
```

---

### 2. Submit Evaluation Event

Submit a single production event for evaluation.

**Endpoint:** `POST /v1/eval/events`

**Request Body:**
```json
{
  "event_id": "evt_abc123",
  "suite_name": "production_rag",
  "input": {
    "query": "What is the capital of France?",
    "context": [
      {
        "text": "Paris is the capital of France.",
        "source_id": "doc_123"
      }
    ]
  },
  "output": {
    "answer": "Paris is the capital of France.",
    "citations": ["doc_123"],
    "latency_ms": 250
  },
  "reference": {
    "answer": "Paris"
  },
  "metadata": {
    "user_id": "user_456",
    "session_id": "session_789",
    "timestamp": "2025-11-05T10:00:00Z"
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:8080/v1/eval/events \
  -H "Content-Type: application/json" \
  -d '{
    "event_id": "evt_abc123",
    "suite_name": "production_rag",
    "input": {
      "query": "What is the capital of France?",
      "context": [
        {"text": "Paris is the capital of France.", "source_id": "doc_123"}
      ]
    },
    "output": {
      "answer": "Paris is the capital of France.",
      "citations": ["doc_123"]
    },
    "reference": {"answer": "Paris"},
    "metadata": {
      "user_id": "user_456",
      "session_id": "session_789"
    }
  }'
```

**Response (202 Accepted):**
```json
{
  "event_id": "evt_abc123",
  "queued_at": "2025-11-05T10:00:00Z",
  "estimated_processing_time": "1-5 minutes",
  "queue_position": 1
}
```

**Field Descriptions:**
- `event_id` (optional): Unique identifier. Auto-generated if not provided.
- `suite_name` (optional): Evaluation suite to use. Defaults to "production_monitoring".
- `input` (required): Original input to your system (query, context, etc.)
- `output` (required): System's response to evaluate
- `reference` (optional): Ground truth for comparison metrics
- `metadata` (optional): Additional context for filtering/analysis

---

### 3. Batch Event Ingestion

Submit up to 100 events in a single request for efficient batch processing.

**Endpoint:** `POST /v1/eval/events/batch`

**Request Body:**
```json
{
  "events": [
    {
      "input": {"query": "Question 1"},
      "output": {"answer": "Answer 1"}
    },
    {
      "input": {"query": "Question 2"},
      "output": {"answer": "Answer 2"}
    }
  ]
}
```

**Example:**
```bash
curl -X POST http://localhost:8080/v1/eval/events/batch \
  -H "Content-Type: application/json" \
  -d '{
    "events": [
      {
        "input": {"query": "What is ML?"},
        "output": {"answer": "Machine Learning is..."}
      },
      {
        "input": {"query": "What is AI?"},
        "output": {"answer": "Artificial Intelligence is..."}
      }
    ]
  }'
```

**Response (202 Accepted):**
```json
{
  "batch_id": "batch_xyz789",
  "accepted_count": 95,
  "rejected_count": 5,
  "rejected_events": [
    {
      "index": 3,
      "reason": "Missing required field: output"
    },
    {
      "index": 7,
      "reason": "Invalid input format"
    }
  ]
}
```

**Limits:**
- Maximum 100 events per batch request
- Events exceeding this limit will be rejected with 400 Bad Request

---

### 4. Check Event Status

Query the processing status of a submitted event.

**Endpoint:** `GET /v1/eval/status/{event_id}`

**Example:**
```bash
curl http://localhost:8080/v1/eval/status/evt_abc123
```

**Response (200 OK):**
```json
{
  "event_id": "evt_abc123",
  "status": "completed",
  "queued_at": "2025-11-05T10:00:00Z",
  "started_at": "2025-11-05T10:01:00Z",
  "completed_at": "2025-11-05T10:02:30Z",
  "results_available": true,
  "error_message": null
}
```

**Status Values:**
- `queued`: Event accepted, waiting for processing
- `processing`: Currently being evaluated
- `completed`: Evaluation finished successfully
- `failed`: Evaluation encountered an error

---

### 5. Retrieve Evaluation Report

Get the complete evaluation report for a run.

**Endpoint:** `GET /v1/reports/{run_id}`

**Query Parameters:**
- `format` (optional): Response format - `json` (default) or `html`

**Example (JSON):**
```bash
curl http://localhost:8080/v1/reports/run_12345
```

**Example (HTML):**
```bash
curl http://localhost:8080/v1/reports/run_12345?format=html
```

**Response (200 OK - JSON):**
```json
{
  "run_id": "run_12345",
  "suite_name": "production_rag",
  "created_at": "2025-11-05T10:00:00Z",
  "status": "complete",
  "results": [...],
  "aggregates": {
    "faithfulness": 0.85,
    "answer_relevancy": 0.78
  },
  "metadata": {...}
}
```

---

## Error Handling

All error responses follow a consistent format:

```json
{
  "error": "error_type",
  "message": "Human-readable error description",
  "details": [...],
  "timestamp": "2025-11-05T10:00:00Z"
}
```

### HTTP Status Codes

- `200 OK`: Request successful
- `202 Accepted`: Event queued for processing
- `400 Bad Request`: Invalid request data (validation failed)
- `404 Not Found`: Event or report not found
- `429 Too Many Requests`: Rate limit exceeded (if configured)
- `500 Internal Server Error`: Server-side error

### Common Error Examples

**Missing Required Field:**
```json
{
  "error": "validation_error",
  "message": "Invalid request data",
  "details": [
    {
      "loc": ["output"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ],
  "timestamp": "2025-11-05T10:00:00Z"
}
```

**Event Not Found:**
```json
{
  "error": "http_error",
  "message": "Event not found",
  "timestamp": "2025-11-05T10:00:00Z"
}
```

---

## Integration Examples

### Python (httpx)
```python
import httpx

async def submit_evaluation():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8080/v1/eval/events",
            json={
                "input": {"query": "What is RAG?"},
                "output": {"answer": "Retrieval Augmented Generation..."}
            }
        )
        event = response.json()
        print(f"Event queued: {event['event_id']}")
        
        # Check status
        status_response = await client.get(
            f"http://localhost:8080/v1/eval/status/{event['event_id']}"
        )
        print(status_response.json())
```

### Node.js (fetch)
```javascript
async function submitEvaluation() {
  const response = await fetch('http://localhost:8080/v1/eval/events', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      input: {query: 'What is RAG?'},
      output: {answer: 'Retrieval Augmented Generation...'}
    })
  });
  
  const event = await response.json();
  console.log(`Event queued: ${event.event_id}`);
}
```

### Production Monitoring Pattern
```python
from ulei_client import ULEIMonitor

# Initialize monitor
monitor = ULEIMonitor(
    server_url="http://localhost:8080",
    suite_name="production_rag"
)

# In your application
@app.route("/query")
async def handle_query(query: str):
    # Get your system's response
    response = await rag_system.query(query)
    
    # Submit for shadow evaluation (non-blocking)
    await monitor.submit_event(
        input={"query": query},
        output={"answer": response.answer},
        metadata={"user_id": request.user_id}
    )
    
    return response
```

---

## Performance Considerations

### Throughput
- Batch ingestion is more efficient for high-volume scenarios
- Use multiple workers (`--workers 4`) for production deployments
- Events are processed asynchronously in background tasks

### Resource Usage
- Queue size is tracked and reported in health checks
- Configure evaluation suite timeouts to prevent resource exhaustion
- Consider rate limiting for public-facing deployments

### Monitoring
- Health endpoint provides real-time queue status
- Enable access logs for request tracking (`--access-log`)
- Metrics export available via Prometheus reporter

---

## See Also

- [Getting Started Guide](GETTING_STARTED.md) - Setup and configuration
- [Quick Reference](QUICK_REFERENCE.md) - Command-line usage
- [OpenAPI Spec](../specs/001-unified-eval-interface/contracts/http-api.yaml) - Complete API contract
