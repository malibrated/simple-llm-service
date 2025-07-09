# LLM Service API Reference

This document provides comprehensive API documentation for the LLM Service, including all endpoints, request/response formats, and client integration examples.

## Table of Contents

1. [Service Discovery](#service-discovery)
2. [Starting the Service](#starting-the-service)
3. [API Endpoints](#api-endpoints)
   - [Health Check](#health-check)
   - [List Models](#list-models)
   - [Chat Completions](#chat-completions)
   - [Text Completions](#text-completions)
   - [Embeddings](#embeddings)
   - [Reranking](#reranking)
4. [Structured Output](#structured-output)
5. [Error Handling](#error-handling)
6. [Client Libraries](#client-libraries)
7. [Examples](#examples)

## Service Discovery

The LLM Service uses dynamic port allocation with a discovery mechanism to avoid port conflicts.

### Port Discovery

When the service starts, it writes its port number to a `.port` file in the service directory:

```bash
# Default location
/Users/patrickpark/Documents/Work/utils/llmservice/.port
```

### Reading the Port

#### Python
```python
from pathlib import Path

def get_service_url():
    port_file = Path("/Users/patrickpark/Documents/Work/utils/llmservice/.port")
    if port_file.exists():
        port = int(port_file.read_text().strip())
        return f"http://localhost:{port}"
    # Fallback to default
    return "http://localhost:8000"
```

#### Bash
```bash
PORT=$(cat /Users/patrickpark/Documents/Work/utils/llmservice/.port 2>/dev/null || echo "8000")
BASE_URL="http://localhost:$PORT"
```

#### JavaScript/TypeScript
```javascript
const fs = require('fs');
const path = require('path');

function getServiceUrl() {
    const portFile = path.join('/Users/patrickpark/Documents/Work/utils/llmservice', '.port');
    try {
        const port = fs.readFileSync(portFile, 'utf8').trim();
        return `http://localhost:${port}`;
    } catch {
        return 'http://localhost:8000';
    }
}
```

## Starting the Service

### Using the Startup Script (Recommended)

The `start_service.sh` script handles virtual environment activation and environment setup:

```bash
cd /Users/patrickpark/Documents/Work/utils/llmservice
./start_service.sh
```

Features of the startup script:
- Automatically activates the virtual environment
- Sets up required environment variables (e.g., `KMP_DUPLICATE_LIB_OK` for OpenMP)
- Starts the service with proper Python path
- Handles graceful shutdown

### Manual Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Set environment variables
export KMP_DUPLICATE_LIB_OK=TRUE

# Start service
python server.py

# Or with custom host/port
PORT=8080 HOST=0.0.0.0 python server.py
```

### Environment Variables

Key environment variables for service configuration:

```bash
# Service Configuration
HOST=127.0.0.1              # Default: 127.0.0.1
PORT=8000                   # Default: 0 (auto-assign)
ENABLE_CORS=true           # Default: true
AUTO_SHUTDOWN_MINUTES=30   # Default: 30 (0 to disable)

# Logging
LOG_LEVEL=INFO             # Options: DEBUG, INFO, WARNING, ERROR

# Cache Configuration
ENABLE_CACHE=true          # Default: true
CACHE_MAX_SIZE=1000       # Default: 1000
CACHE_TTL_SECONDS=3600    # Default: 3600
```

## API Endpoints

### Base URL

All API endpoints are relative to the base URL:
```
http://localhost:{port}
```

### Health Check

Check if the service is running and healthy.

**Endpoint:** `GET /health`

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2024-03-20T15:30:45.123456"
}
```

### List Models

Get available models and their current status.

**Endpoint:** `GET /v1/models`

**Response:**
```json
{
    "object": "list",
    "data": [
        {
            "id": "light/Qwen2.5-0.5B-Instruct-Q8_0",
            "object": "model",
            "created": 1710950400,
            "owned_by": "llmservice-light",
            "permission": [],
            "root": "light/Qwen2.5-0.5B-Instruct-Q8_0",
            "parent": null
        },
        {
            "id": "medium/gemma-3-4b-it-qat-4bit (loaded)",
            "object": "model",
            "created": 1710950400,
            "owned_by": "llmservice-medium",
            "permission": [],
            "root": "medium/gemma-3-4b-it-qat-4bit (loaded)",
            "parent": null
        }
    ]
}
```

**Note:** Models with "(loaded)" suffix are currently loaded in memory.

### Chat Completions

OpenAI-compatible chat completions endpoint.

**Endpoint:** `POST /v1/chat/completions`

**Request Body:**
```json
{
    "model": "medium",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_tokens": 150,
    "stream": false,
    "seed": 42,
    "stop": ["\n\n"],
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "response_format": {
        "type": "json_object"
    }
}
```

**Parameters:**
- `model` (required): Model tier ("light", "medium", "heavy") or specific model ID
- `messages` (required): Array of message objects with `role` and `content`
- `temperature` (optional): Sampling temperature (0.0-2.0)
- `top_p` (optional): Nucleus sampling (0.0-1.0)
- `top_k` (optional): Top-k sampling
- `max_tokens` (optional): Maximum tokens to generate
- `stream` (optional): Enable streaming response
- `seed` (optional): Random seed for reproducibility
- `stop` (optional): Stop sequences
- `response_format` (optional): Structured output format

**Response (Non-streaming):**
```json
{
    "id": "chatcmpl-123456",
    "object": "chat.completion",
    "created": 1710950400,
    "model": "medium",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The capital of France is Paris.",
                "refusal": null
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 8,
        "total_tokens": 28
    },
    "system_fingerprint": null
}
```

**Response (Streaming):**
```
data: {"id":"chatcmpl-123456","object":"chat.completion.chunk","created":1710950400,"model":"medium","choices":[{"index":0,"delta":{"role":"assistant","content":"The"},"finish_reason":null}]}

data: {"id":"chatcmpl-123456","object":"chat.completion.chunk","created":1710950400,"model":"medium","choices":[{"index":0,"delta":{"content":" capital"},"finish_reason":null}]}

data: [DONE]
```

### Text Completions

Legacy OpenAI-compatible text completions endpoint.

**Endpoint:** `POST /v1/completions`

**Request Body:**
```json
{
    "model": "light",
    "prompt": "The quick brown fox",
    "max_tokens": 50,
    "temperature": 0.7,
    "top_p": 0.95,
    "n": 1,
    "stream": false,
    "logprobs": null,
    "stop": ["\n"]
}
```

**Response:**
```json
{
    "id": "cmpl-123456",
    "object": "text_completion",
    "created": 1710950400,
    "model": "light",
    "choices": [
        {
            "text": " jumps over the lazy dog.",
            "index": 0,
            "logprobs": null,
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 4,
        "completion_tokens": 6,
        "total_tokens": 10
    }
}
```

### Embeddings

Generate embeddings using BGE-M3 or other embedding models.

**Endpoint:** `POST /v1/embeddings`

**Request Body:**
```json
{
    "model": "embedding",
    "input": "The quick brown fox jumps over the lazy dog",
    "encoding_format": "float",
    "dimensions": 1024,
    "embedding_type": "dense",
    "return_sparse": false
}
```

**Parameters:**
- `model` (required): "embedding" or specific embedding model ID
- `input` (required): Text string or array of strings
- `encoding_format` (optional): "float" or "base64"
- `dimensions` (optional): Embedding dimension (if model supports truncation)
- `embedding_type` (optional): "dense", "sparse", or "colbert" (BGE-M3 specific)
- `return_sparse` (optional): Return sparse embeddings alongside dense (BGE-M3)

**Response:**
```json
{
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "index": 0,
            "embedding": [0.1234, -0.5678, ...],
            "sparse_embedding": {
                "indices": [102, 5423, ...],
                "values": [0.89, 0.76, ...]
            }
        }
    ],
    "model": "embedding/bge-m3-q8_0",
    "usage": {
        "prompt_tokens": 9,
        "total_tokens": 9
    }
}
```

### Reranking

Rerank documents based on relevance to a query.

**Endpoint:** `POST /v1/rerank`

**Request Body:**
```json
{
    "model": "reranker",
    "query": "What is machine learning?",
    "documents": [
        "Machine learning is a subset of AI...",
        "The weather today is sunny...",
        "Deep learning uses neural networks..."
    ],
    "top_n": 2,
    "return_documents": true
}
```

**Parameters:**
- `model` (required): "reranker" or specific reranker model ID
- `query` (required): The search query
- `documents` (required): Array of document strings
- `top_n` (optional): Number of top documents to return
- `return_documents` (optional): Include original documents in response

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
            "score": 0.87,
            "document": "Deep learning uses neural networks..."
        }
    ],
    "model": "reranker/bge-reranker-v2-m3-q8_0",
    "usage": {
        "prompt_tokens": 45,
        "total_tokens": 45
    }
}
```

## Structured Output

The service supports OpenAI-compatible structured output for generating valid JSON.

### JSON Object Mode

Force the model to output valid JSON:

```json
{
    "model": "medium",
    "messages": [
        {
            "role": "user",
            "content": "List 3 programming languages with their year of creation"
        }
    ],
    "response_format": {
        "type": "json_object"
    },
    "temperature": 0.1
}
```

**Response:**
```json
{
    "choices": [
        {
            "message": {
                "content": "{\"languages\": [{\"name\": \"Python\", \"year\": 1991}, {\"name\": \"JavaScript\", \"year\": 1995}, {\"name\": \"Go\", \"year\": 2009}]}"
            }
        }
    ]
}
```

### JSON Schema Mode (Future)

Constrain output to a specific schema:

```json
{
    "model": "medium",
    "messages": [...],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "strict": true,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "number"},
                    "email": {"type": "string", "format": "email"}
                },
                "required": ["name", "age"]
            }
        }
    }
}
```

## Error Handling

The service returns standard HTTP status codes and error messages.

### Error Response Format

```json
{
    "error": {
        "message": "Model not found: invalid_model",
        "type": "invalid_request_error",
        "code": "model_not_found"
    }
}
```

### Common Status Codes

- `200 OK`: Successful request
- `400 Bad Request`: Invalid parameters or request format
- `404 Not Found`: Model or endpoint not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Model loading or capacity issues

## Auto-Start Feature

The Python client library includes an auto-start feature that can automatically start the LLM service if it's not running:

```python
from examples.llm_service_client import LLMServiceClient

# Enable auto-start
client = LLMServiceClient(auto_start=True)

# Or manually start the service
client = LLMServiceClient()
await client.start_service()

# Quick functions also support auto-start
from examples.llm_service_client import quick_chat
response = await quick_chat("Hello!", auto_start=True)
```

## Client Libraries

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8327/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="medium",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
```

### Langchain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8327/v1",
    api_key="not-needed",
    model="heavy",
    temperature=0.7
)
```

### JavaScript/TypeScript

```javascript
import OpenAI from 'openai';

const openai = new OpenAI({
    baseURL: 'http://localhost:8327/v1',
    apiKey: 'not-needed',
});

const response = await openai.chat.completions.create({
    model: 'medium',
    messages: [{ role: 'user', content: 'Hello!' }],
});
```

### cURL

```bash
curl -X POST http://localhost:8327/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "medium",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Examples

### Basic Chat

```python
import httpx
import asyncio

async def chat_example():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8327/v1/chat/completions",
            json={
                "model": "medium",
                "messages": [
                    {"role": "user", "content": "Explain quantum computing in simple terms"}
                ],
                "max_tokens": 200
            }
        )
        result = response.json()
        print(result["choices"][0]["message"]["content"])

asyncio.run(chat_example())
```

### Streaming Response

```python
import httpx
import asyncio
import json

async def stream_example():
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8327/v1/chat/completions",
            json={
                "model": "medium",
                "messages": [{"role": "user", "content": "Write a haiku"}],
                "stream": True
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if "content" in chunk["choices"][0]["delta"]:
                        print(chunk["choices"][0]["delta"]["content"], end="")

asyncio.run(stream_example())
```

### Structured Output

```python
import httpx
import asyncio
import json

async def structured_output_example():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8327/v1/chat/completions",
            json={
                "model": "medium",
                "messages": [
                    {"role": "user", "content": "Generate a user profile with name, age, and hobbies"}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1
            }
        )
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        profile = json.loads(content)
        print(json.dumps(profile, indent=2))

asyncio.run(structured_output_example())
```

### Embeddings with BGE-M3

```python
import httpx
import asyncio
import numpy as np

async def embedding_example():
    async with httpx.AsyncClient() as client:
        # Generate embeddings
        response = await client.post(
            "http://localhost:8327/v1/embeddings",
            json={
                "model": "embedding",
                "input": [
                    "Machine learning is fascinating",
                    "I love pizza",
                    "Neural networks are powerful"
                ],
                "embedding_type": "dense"
            }
        )
        result = response.json()
        
        # Extract embeddings
        embeddings = [item["embedding"] for item in result["data"]]
        
        # Calculate similarities
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # Compare first text with others
        for i in range(1, len(embeddings)):
            sim = cosine_similarity(embeddings[0], embeddings[i])
            print(f"Similarity between text 0 and {i}: {sim:.3f}")

asyncio.run(embedding_example())
```

### Document Reranking

```python
import httpx
import asyncio

async def reranking_example():
    async with httpx.AsyncClient() as client:
        documents = [
            "Python is a high-level programming language.",
            "The weather today is sunny and warm.",
            "Machine learning models can be trained with Python.",
            "I enjoy cooking Italian food.",
            "TensorFlow and PyTorch are popular ML frameworks."
        ]
        
        response = await client.post(
            "http://localhost:8327/v1/rerank",
            json={
                "model": "reranker",
                "query": "programming with Python for machine learning",
                "documents": documents,
                "top_n": 3,
                "return_documents": True
            }
        )
        
        result = response.json()
        print("Top relevant documents:")
        for item in result["data"]:
            print(f"Score: {item['score']:.3f} - {item['document']}")

asyncio.run(reranking_example())
```

## Performance Considerations

1. **Model Loading**: First request to a model triggers loading (2-10 seconds)
2. **Caching**: Repeated identical requests are served from cache
3. **Concurrency**: 
   - llama.cpp models: Full concurrent processing
   - MLX models: Sequential processing per tier (Metal limitation)
4. **Memory**: Monitor memory usage, especially with multiple large models
5. **Auto-shutdown**: Service shuts down after inactivity to free resources

## Security Notes

- The service does not require authentication by default
- Intended for local development or trusted networks
- For production use, add authentication middleware
- Consider using a reverse proxy (nginx, caddy) for HTTPS