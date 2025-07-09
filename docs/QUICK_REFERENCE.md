# LLM Service Quick Reference

## Starting the Service

```bash
# Start with automatic port selection
./start_service.sh

# Check which port was assigned
cat .port
```

## Port Discovery

### Python
```python
port = int(open(".port").read().strip())
base_url = f"http://localhost:{port}"
```

### Bash
```bash
PORT=$(cat .port)
curl http://localhost:$PORT/health
```

### JavaScript
```javascript
const port = require('fs').readFileSync('.port', 'utf8').trim();
const baseUrl = `http://localhost:${port}`;
```

## Common API Calls

### Chat Completion
```bash
curl -X POST http://localhost:$(cat .port)/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "medium",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Structured Output (JSON)
```bash
curl -X POST http://localhost:$(cat .port)/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "medium",
    "messages": [{"role": "user", "content": "List 3 colors"}],
    "response_format": {"type": "json_object"},
    "temperature": 0.1
  }'
```

### List Models
```bash
curl http://localhost:$(cat .port)/v1/models | jq .
```

## Python Quick Start

```python
import httpx
import asyncio

async def main():
    # Get port
    port = int(open(".port").read().strip())
    base_url = f"http://localhost:{port}"
    
    # Make request
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "medium",
                "messages": [{"role": "user", "content": "Hello!"}]
            }
        )
        result = response.json()
        print(result["choices"][0]["message"]["content"])

asyncio.run(main())
```

## Using the Client Library

```python
from examples.llm_service_client import LLMServiceClient

async def main():
    # Auto-start service if not running
    client = LLMServiceClient(auto_start=True)
    
    # Simple chat
    response = await client.chat("What is Python?")
    
    # Structured output
    data = await client.structured_chat("Generate a user profile")
    
    # Streaming
    async for chunk in client.chat_stream("Tell me a story"):
        print(chunk, end="")

asyncio.run(main())

# Or use quick functions (auto-start by default)
from examples.llm_service_client import quick_chat
response = await quick_chat("Hello!")
```

## OpenAI SDK Compatibility

```python
from openai import OpenAI

# Point to local service
client = OpenAI(
    base_url=f"http://localhost:{open('.port').read().strip()}/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="medium",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Model Tiers

- **light**: Fast responses, simple tasks
- **medium**: Balanced performance, general use
- **heavy**: Complex reasoning, detailed outputs
- **embedding**: Text embeddings (BGE-M3)
- **reranker**: Document reranking

## Environment Variables

```bash
# Service
PORT=8000                    # Fixed port (0 = auto)
HOST=127.0.0.1              # Bind address
AUTO_SHUTDOWN_MINUTES=30    # Auto shutdown timer

# Models
LIGHT_MODEL_PATH=/path/to/model.gguf
MEDIUM_MODEL_PATH=/path/to/model.gguf
HEAVY_MODEL_PATH=/path/to/model.gguf

# Parameters (per tier)
{TIER}_TEMPERATURE=0.7
{TIER}_MAX_TOKENS=1000
{TIER}_N_CTX=8192
{TIER}_N_GPU_LAYERS=-1
```

## Troubleshooting

### Service won't start
```bash
# Check if port is in use
lsof -i :$(cat .port 2>/dev/null || echo 8000)

# Check logs
python server.py  # Run directly to see errors
```

### Model not loading
```bash
# Verify model path
ls -la $MEDIUM_MODEL_PATH

# Check permissions
chmod +r $MEDIUM_MODEL_PATH
```

### Slow performance
```bash
# Enable GPU
export MEDIUM_N_GPU_LAYERS=-1

# Increase threads
export MEDIUM_N_THREADS=$(sysctl -n hw.ncpu)
```

### Memory issues
```bash
# Reduce context size
export HEAVY_N_CTX=2048

# Use quantized models (Q4_K_M, Q5_K_M)
```