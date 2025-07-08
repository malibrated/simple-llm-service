# LLM Service

A lightweight, performant REST API service for Large Language Models with OpenAI-compatible endpoints. Supports both llama.cpp (GGUF) and Apple MLX models with configurable parameters and intelligent caching.

## Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API endpoints
- **Multi-Backend Support**: Works with both llama.cpp (.gguf) and MLX models
- **Model Tiers**: Organize models into LIGHT, MEDIUM, and HEAVY categories
- **Lazy Model Loading**: Models are loaded on-demand for faster startup times
- **Configurable Parameters**: All LLM parameters configurable via environment variables
- **Response Caching**: In-memory caching with optional disk persistence
- **Auto-Shutdown**: Automatic service shutdown after inactivity period
- **Langchain Compatible**: Works seamlessly with Langchain and Langgraph
- **Structured Output**: Support for JSON grammars and structured generation
- **Async Architecture**: High-performance async request handling

> **⚠️ MLX Compatibility Notice**: Due to a parameter passing issue in MLX-LM v0.25.3, temperature and sampling parameters are not supported for MLX models. We recommend using llama.cpp models until this is resolved. See [Known Issues](#known-issues) for details.

## Installation

```bash
# Clone the repository
cd /Users/patrickpark/Documents/Work/utils/llmservice

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For MLX support (macOS with Apple Silicon)
pip install mlx mlx-lm
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` to configure your models:
```env
# Example configuration for three tiers
LIGHT_MODEL_PATH=/path/to/qwen-0.5b.gguf
LIGHT_BACKEND=llamacpp
LIGHT_TEMPERATURE=0.1
LIGHT_MAX_TOKENS=1000

MEDIUM_MODEL_PATH=/path/to/gemma-4b.gguf
MEDIUM_BACKEND=llamacpp
MEDIUM_TEMPERATURE=0.3
MEDIUM_MAX_TOKENS=2048

HEAVY_MODEL_PATH=/path/to/mistral-24b.gguf
HEAVY_BACKEND=llamacpp
HEAVY_TEMPERATURE=0.7
HEAVY_MAX_TOKENS=4096
```

## Usage

### Starting the Service

```bash
# Run directly
python server.py

# Or with uvicorn for development
uvicorn server:app --reload

# Custom host/port
PORT=8080 HOST=127.0.0.1 python server.py
```

### API Endpoints

#### Chat Completions (OpenAI Compatible)
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "medium",  # or "light", "heavy", or specific model name
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
)
```

#### Text Completions
```python
response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "light",
        "prompt": "The capital of France is",
        "max_tokens": 50,
        "temperature": 0.1
    }
)
```

#### List Available Models
```python
response = requests.get("http://localhost:8000/v1/models")
```

### Using with Langchain

```python
from langchain_openai import ChatOpenAI

# Configure to use local service
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # API key not required for local service
    model="medium",
    temperature=0.3
)

# Use as normal
response = llm.invoke("Tell me about quantum computing")
```

### Using with Langgraph

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import Graph

# Create LLM instance
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="heavy"
)

# Use in your graph
graph = Graph()
# ... configure your graph with the LLM
```

## Advanced Features

### Structured Output with JSON Grammar

```python
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "medium",
        "messages": [{"role": "user", "content": "Extract entities from: Apple Inc. was founded by Steve Jobs."}],
        "response_format": {"type": "json_object"},
        "grammar": '''
root ::= object
object ::= "{" ws "\\"entities\\"" ws ":" ws array ws "}"
array ::= "[" ws entity ("," ws entity)* ws "]"
entity ::= "{" ws "\\"name\\"" ws ":" ws string ws "," ws "\\"type\\"" ws ":" ws string ws "}"
string ::= "\\"" ([^"\\\\] | "\\\\" .)* "\\""
ws ::= [ \\t\\n]*
'''
    }
)
```

### Model-Specific Parameters

Each tier can have different default parameters:

```env
# Fast responses for classification
LIGHT_TEMPERATURE=0.1
LIGHT_MAX_TOKENS=100
LIGHT_TOP_K=10

# Balanced for general use
MEDIUM_TEMPERATURE=0.5
MEDIUM_MAX_TOKENS=1000
MEDIUM_TOP_P=0.95

# Creative generation
HEAVY_TEMPERATURE=0.8
HEAVY_MAX_TOKENS=4096
HEAVY_REPEAT_PENALTY=1.2
```

### Caching Configuration

```env
# Enable response caching
ENABLE_CACHE=true
CACHE_MAX_SIZE=1000
CACHE_TTL_SECONDS=3600

# Optional disk persistence
CACHE_PERSIST_TO_DISK=true
CACHE_DIR=.cache/responses
```

## Performance Tips

1. **Lazy Loading**: Models are loaded on first use, reducing startup time
2. **Request Queuing**: Each tier has its own processing queue
3. **Auto-shutdown**: Service shuts down after inactivity to save resources
4. **GPU Acceleration**: Set `N_GPU_LAYERS=-1` to use all GPU layers
5. **Thread Optimization**: Adjust `N_THREADS` based on your CPU cores

### Lazy Loading Behavior

The service implements lazy loading for all models:
- **Fast Startup**: Service starts immediately without loading any models
- **On-Demand Loading**: Models are loaded when first requested
- **Memory Efficient**: Only models that are actually used consume memory
- **Status Tracking**: The `/v1/models` endpoint shows which models are loaded

Example startup behavior:
```bash
# Service starts instantly
$ python server.py
INFO: Lazy loading enabled for 3 model configurations
INFO: LLM Service ready

# First request to 'light' model triggers loading
$ curl -X POST http://localhost:8000/v1/chat/completions -d '{"model": "light", ...}'
INFO: Lazy loading light model on first use...
INFO: Loaded light model in 2.45s

# Subsequent requests use the loaded model immediately
```

## Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### Cache Statistics
```python
# Available through internal API (not OpenAI compatible)
response = requests.get("http://localhost:8000/internal/cache/stats")
```

## Troubleshooting

### Common Issues

1. **Model not loading**: Check file path and permissions in `.env`
2. **Out of memory**: Reduce `N_CTX` or use quantized models
3. **Slow generation**: Enable GPU layers with `N_GPU_LAYERS=-1`
4. **Cache misses**: Increase `CACHE_MAX_SIZE` or `CACHE_TTL_SECONDS`

### Debug Mode

Enable debug logging:
```env
LOG_LEVEL=DEBUG
```

## Known Issues

### MLX Parameter Compatibility (mlx-lm v0.25.3)

There is currently an issue with MLX-LM where the `generate` function passes unsupported parameters to internal functions, causing a `TypeError: generate_step() got an unexpected keyword argument 'temp'` error. 

**Impact**: Temperature, top_p, and other sampling parameters are ignored for MLX models.

**Workaround**: Use llama.cpp models instead of MLX models for full parameter support.

**Status**: We've implemented a workaround that allows MLX models to function, but without sampling parameter support. An issue has been reported to the MLX-LM developers.

## License

This project is designed for internal use. Please ensure you comply with the licenses of any models you use with this service.