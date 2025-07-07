# LLM Service API Documentation

A high-performance, OpenAI-compatible REST API service for Large Language Models (LLMs), embedding models, and reranking models. Supports multiple model tiers, backends (llama.cpp and MLX), and advanced features like BGE-M3's dense/sparse embeddings.

## Table of Contents
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Request/Response Formats](#requestresponse-formats)
- [Advanced Features](#advanced-features)
- [Client Examples](#client-examples)
- [Error Handling](#error-handling)

## Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd llmservice

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For MLX support on Apple Silicon
pip install mlx mlx-lm

# Copy and configure environment
cp .env.example .env
# Edit .env with your model paths
```

### Starting the Service
```bash
# Using the startup script
./start_service.sh

# Or directly with Python
python server.py

# With custom port
PORT=8080 python server.py
```

### Basic Usage
```python
import requests

# Check service health
response = requests.get("http://localhost:8000/health")

# Generate text
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "medium",
    "messages": [{"role": "user", "content": "Hello!"}]
})
```

## Configuration

### Environment Variables

The service is configured via `.env` file with five model tiers:

#### Server Configuration
```bash
PORT=8000                    # API port
HOST=0.0.0.0                # Host address
LOG_LEVEL=INFO              # Logging level
SHUTDOWN_TIMEOUT=600        # Auto-shutdown after inactivity (seconds)
```

#### Cache Configuration
```bash
ENABLE_CACHE=true           # Enable response caching
CACHE_MAX_SIZE=1000        # Maximum cache entries
CACHE_TTL_SECONDS=3600     # Cache time-to-live
CACHE_PERSIST_TO_DISK=false # Persist cache to disk
```

#### Model Tiers

Each tier (`LIGHT`, `MEDIUM`, `HEAVY`, `EMBEDDING`, `RERANKER`) has these settings:

```bash
{TIER}_MODEL_PATH=/path/to/model     # Model file path
{TIER}_BACKEND=llamacpp              # Backend: llamacpp or mlx
{TIER}_N_CTX=8192                    # Context window size
{TIER}_N_BATCH=512                   # Batch size (llamacpp only)
{TIER}_N_THREADS=8                   # CPU threads
{TIER}_N_GPU_LAYERS=-1               # GPU layers (-1 for all)
{TIER}_TEMPERATURE=0.7               # Default temperature
{TIER}_TOP_P=0.95                    # Default top-p
{TIER}_TOP_K=40                      # Default top-k
{TIER}_MAX_TOKENS=1000               # Default max tokens
{TIER}_REPEAT_PENALTY=1.1            # Repetition penalty
{TIER}_USE_MMAP=true                 # Memory mapping
{TIER}_USE_MLOCK=false               # Lock in memory
{TIER}_EMBEDDING_MODE=false          # Enable embedding mode
{TIER}_EMBEDDING_DIMENSION=1024      # Embedding dimensions
```

### Port Discovery

The service writes its port to `.port` file on startup:

```python
# Discover service port
from pathlib import Path

port_file = Path(".port")
if port_file.exists():
    port = int(port_file.read_text().strip())
else:
    port = 8000  # Default
```

## API Endpoints

### GET /health
Health check endpoint.

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2024-01-01T00:00:00"
}
```

### GET /v1/models
List available models.

**Response:**
```json
{
    "object": "list",
    "data": [
        {
            "id": "light/qwen-0.5b",
            "object": "model",
            "owned_by": "llmservice-light"
        },
        {
            "id": "embedding/bge-m3",
            "object": "model",
            "owned_by": "llmservice-embedding"
        }
    ]
}
```

### POST /v1/chat/completions
Generate chat completions (OpenAI-compatible).

**Request:**
```json
{
    "model": "medium",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100,
    "stream": false
}
```

**Response:**
```json
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "medium",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The capital of France is Paris."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 30
    }
}
```

**Streaming Response:**
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"medium","choices":[{"index":0,"delta":{"content":"The"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"medium","choices":[{"index":0,"delta":{"content":" capital"},"finish_reason":null}]}

data: [DONE]
```

### POST /v1/completions
Text completions (legacy OpenAI format).

**Request:**
```json
{
    "model": "light",
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "temperature": 0.8
}
```

### POST /v1/embeddings
Generate embeddings with BGE-M3 support.

**Request:**
```json
{
    "model": "embedding",
    "input": "The quick brown fox",
    "encoding_format": "float",
    "embedding_type": "dense",
    "return_sparse": false
}
```

**Advanced BGE-M3 Request:**
```json
{
    "model": "embedding",
    "input": ["Text 1", "Text 2"],
    "embedding_type": "dense",
    "return_sparse": true,
    "dimensions": 512
}
```

**Response:**
```json
{
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "index": 0,
            "embedding": [0.023, -0.011, ...],
            "sparse_embedding": {
                "1234": 0.82,
                "5678": 0.31
            }
        }
    ],
    "model": "embedding",
    "usage": {
        "prompt_tokens": 5,
        "total_tokens": 5
    }
}
```

### POST /v1/rerank
Rerank documents based on query relevance.

**Request:**
```json
{
    "model": "reranker",
    "query": "What is machine learning?",
    "documents": [
        "Machine learning is a subset of AI...",
        "The weather today is sunny...",
        "ML algorithms learn from data..."
    ],
    "top_k": 2,
    "return_documents": true
}
```

**Response:**
```json
{
    "object": "list",
    "data": [
        {
            "index": 0,
            "score": 0.95,
            "document": "Machine learning is a subset of AI..."
        },
        {
            "index": 2,
            "score": 0.89,
            "document": "ML algorithms learn from data..."
        }
    ],
    "model": "reranker",
    "usage": {
        "prompt_tokens": 50,
        "total_tokens": 50
    }
}
```

## Request/Response Formats

### Common Parameters

#### Model Selection
- `model`: Can be a tier name (`light`, `medium`, `heavy`, `embedding`, `reranker`) or a specific model name
- Model routing is automatic based on name patterns:
  - `*embed*`, `*bge*`, `*e5*` → EMBEDDING tier
  - `*rerank*`, `*cross-encoder*` → RERANKER tier
  - Size indicators (`0.5b`, `7b`, etc.) → Appropriate tier

#### Generation Parameters
- `temperature`: 0.0-2.0 (default: model-specific)
- `top_p`: 0.0-1.0 (default: 0.95)
- `top_k`: 1+ (default: 40)
- `max_tokens`: Maximum tokens to generate
- `stop`: String or array of stop sequences
- `stream`: Enable streaming responses

### Structured Output

For JSON generation, use response_format:
```json
{
    "model": "medium",
    "messages": [...],
    "response_format": {"type": "json_object"}
}
```

## Advanced Features

### BGE-M3 Embedding Types

BGE-M3 supports three embedding types:

1. **Dense Embeddings** (default)
```json
{
    "embedding_type": "dense"
}
```

2. **Sparse Embeddings**
```json
{
    "embedding_type": "sparse"
}
```

3. **ColBERT Embeddings** (multi-vector)
```json
{
    "embedding_type": "colbert"
}
```

4. **Hybrid (Dense + Sparse)**
```json
{
    "embedding_type": "dense",
    "return_sparse": true
}
```

### Dimension Reduction

Request specific embedding dimensions:
```json
{
    "model": "embedding",
    "input": "Text",
    "dimensions": 256
}
```

### Batch Processing

Process multiple inputs efficiently:
```json
{
    "model": "embedding",
    "input": ["Text 1", "Text 2", "Text 3"]
}
```

## Client Examples

### Python (requests)
```python
import requests

base_url = "http://localhost:8000"

# Chat completion
response = requests.post(f"{base_url}/v1/chat/completions", json={
    "model": "medium",
    "messages": [
        {"role": "user", "content": "Explain quantum computing"}
    ],
    "max_tokens": 200
})
result = response.json()
print(result["choices"][0]["message"]["content"])

# Embeddings
response = requests.post(f"{base_url}/v1/embeddings", json={
    "model": "embedding",
    "input": "Quantum computing uses quantum mechanics",
    "embedding_type": "dense"
})
embedding = response.json()["data"][0]["embedding"]

# Reranking
response = requests.post(f"{base_url}/v1/rerank", json={
    "model": "reranker",
    "query": "quantum computing applications",
    "documents": ["Doc 1", "Doc 2", "Doc 3"],
    "top_k": 2
})
top_docs = response.json()["data"]
```

### Python (OpenAI SDK)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # API key not required for local service
)

# Chat completion
response = client.chat.completions.create(
    model="medium",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)

# Embeddings
response = client.embeddings.create(
    model="embedding",
    input="Your text here"
)
embedding = response.data[0].embedding
```

### JavaScript/Node.js
```javascript
import OpenAI from 'openai';

const openai = new OpenAI({
    baseURL: 'http://localhost:8000/v1',
    apiKey: 'not-needed',
});

// Chat completion
const completion = await openai.chat.completions.create({
    model: 'medium',
    messages: [{ role: 'user', content: 'Hello!' }],
});
console.log(completion.choices[0].message.content);

// Embeddings
const embedding = await openai.embeddings.create({
    model: 'embedding',
    input: 'Your text here',
});
console.log(embedding.data[0].embedding);
```

### Streaming Example
```python
import requests
import json

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "medium",
        "messages": [{"role": "user", "content": "Write a story"}],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        if line.startswith(b'data: '):
            data = line[6:]  # Remove 'data: ' prefix
            if data == b'[DONE]':
                break
            chunk = json.loads(data)
            if chunk['choices'][0]['delta'].get('content'):
                print(chunk['choices'][0]['delta']['content'], end='')
```

### Langchain Integration
```python
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings

# LLM
llm = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="medium"
)
response = llm.invoke("What is the meaning of life?")

# Embeddings
embeddings = OpenAIEmbeddings(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="embedding"
)
vector = embeddings.embed_query("Your text")
```

## Error Handling

### HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Model not found
- `500`: Internal server error
- `503`: Service unavailable (model loading)

### Error Response Format
```json
{
    "error": {
        "message": "Model not found: unknown-model",
        "type": "invalid_request_error",
        "code": "model_not_found"
    }
}
```

### Common Errors

1. **Model Not Available**
```json
{
    "error": {
        "message": "No models available for tier: heavy",
        "type": "model_error"
    }
}
```

2. **Context Length Exceeded**
```json
{
    "error": {
        "message": "Input exceeds maximum context length",
        "type": "invalid_request_error"
    }
}
```

3. **Invalid Parameters**
```json
{
    "error": {
        "message": "temperature must be between 0.0 and 2.0",
        "type": "invalid_request_error"
    }
}
```

## Performance Tips

1. **Use Appropriate Model Tiers**
   - LIGHT: Simple queries, quick responses
   - MEDIUM: Balanced performance
   - HEAVY: Complex reasoning tasks

2. **Enable Caching**
   - Set `ENABLE_CACHE=true` for repeated queries
   - Configure `CACHE_TTL_SECONDS` appropriately

3. **Batch Processing**
   - Send multiple texts in one embedding request
   - Use batch size appropriate for your model

4. **Context Management**
   - Keep prompts within model's context window
   - Use `n_ctx` setting to control memory usage

5. **GPU Acceleration**
   - Set `N_GPU_LAYERS=-1` to use all GPU layers
   - Monitor GPU memory usage

## Monitoring

### Logs
- Location: Console output
- Level: Configured via `LOG_LEVEL`
- Format: `timestamp - module - level - message`

### Health Monitoring
```bash
# Check service health
curl http://localhost:8000/health

# Monitor with watch
watch -n 5 'curl -s http://localhost:8000/health | jq'
```

### Port Discovery
```bash
# Check which port the service is using
cat .port
```

## Security Considerations

1. **Local Service**
   - Designed for local/internal use
   - No built-in authentication
   - Use reverse proxy for public exposure

2. **CORS**
   - Enabled by default for all origins
   - Configure restrictively for production

3. **Input Validation**
   - All inputs are validated
   - Context length limits enforced
   - Parameter bounds checked

4. **Resource Limits**
   - Auto-shutdown on inactivity
   - Memory mapping for large models
   - Queue limits per tier

## Troubleshooting

### Service Won't Start
- Check model paths in `.env`
- Verify model file formats (GGUF for llama.cpp, directory for MLX)
- Check available memory
- Review logs for specific errors

### Slow Performance
- Enable GPU layers: `N_GPU_LAYERS=-1`
- Reduce context size: `N_CTX=4096`
- Use lighter model tier
- Enable caching

### Out of Memory
- Reduce batch size: `N_BATCH=256`
- Lower context window: `N_CTX=2048`
- Use quantized models
- Enable memory mapping: `USE_MMAP=true`

### Model Not Loading
- Verify file path exists
- Check file permissions
- Ensure correct backend (llamacpp vs mlx)
- Verify model format compatibility

## Contributing

When adding new models or features:

1. **New Embedding Models**
   - Extend `BaseEmbedder` in `embedders.py`
   - Implement required methods
   - Update factory function

2. **New Reranking Models**
   - Extend `BaseReranker` in `rerankers.py`
   - Implement rerank method
   - Update factory function

3. **New Backends**
   - Extend `BaseModelWrapper` in `model_manager.py`
   - Implement initialize and generate methods
   - Update model loading logic

## License

[Insert License Information]

## Support

For issues and questions:
- GitHub Issues: [repository-url]/issues
- Documentation: This file and CLAUDE.md