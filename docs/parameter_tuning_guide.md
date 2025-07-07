# LLM Service Parameter Tuning Guide

## Overview

The LLM Service allows **full parameter tuning per request** while models remain loaded in memory. Each request can override the default parameters configured in `.env`, giving different requesters fine-grained control over generation behavior.

## Tunable Parameters

### Generation Parameters

| Parameter | Type | Range | Description | Example Use Case |
|-----------|------|-------|-------------|------------------|
| `temperature` | float | 0.0-2.0 | Controls randomness | 0.1 for extraction, 0.8 for creative |
| `top_p` | float | 0.0-1.0 | Nucleus sampling threshold | 0.95 for balanced output |
| `top_k` | int | ≥1 | Top-k sampling | 10 for focused, 40 for diverse |
| `max_tokens` | int | ≥1 | Maximum response length | 50 for classification, 4096 for analysis |
| `repeat_penalty` | float | ≥0.0 | Penalize repetition | 1.1 to reduce loops |
| `seed` | int | any | Reproducible generation | 42 for consistent results |
| `stop` | str/list | - | Stop sequences | ["\n", "###"] to control output |

### Structured Output Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `grammar` | string | JSON/GBNF grammar | See grammar examples below |
| `response_format` | object | OpenAI-style format | `{"type": "json_object"}` |

### Model Selection

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `model` | string | Model tier or name | `"light"`, `"medium"`, `"heavy"` |

## Request Examples

### 1. Basic Parameter Tuning

```python
import requests

# Low temperature for factual extraction
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "medium",
        "messages": [{"role": "user", "content": "Extract key facts from this text..."}],
        "temperature": 0.1,
        "max_tokens": 500,
        "top_k": 10  # Very focused
    }
)

# Higher temperature for creative generation
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "heavy",
        "messages": [{"role": "user", "content": "Write a creative legal hypothetical..."}],
        "temperature": 0.8,
        "max_tokens": 2000,
        "top_p": 0.95,
        "repeat_penalty": 1.2  # Avoid repetitive scenarios
    }
)
```

### 2. Using with Langchain

```python
from langchain_openai import ChatOpenAI

# Each instance can have different parameters
classifier = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="light",
    temperature=0.0,  # Deterministic
    max_tokens=20,
    top_k=1  # Greedy decoding
)

extractor = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="medium",
    temperature=0.2,
    max_tokens=1000,
    model_kwargs={
        "repeat_penalty": 1.1,
        "seed": 42  # Reproducible extractions
    }
)

analyzer = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="heavy",
    temperature=0.5,
    max_tokens=4000,
    top_p=0.9
)
```

### 3. Dynamic Parameter Adjustment

```python
class AdaptiveLLM:
    """Dynamically adjust parameters based on task."""
    
    def __init__(self, base_url="http://localhost:8000/v1"):
        self.base_url = base_url
    
    def generate(self, prompt, task_type="general"):
        # Different parameters for different tasks
        params = {
            "classification": {
                "model": "light",
                "temperature": 0.0,
                "max_tokens": 10,
                "top_k": 1
            },
            "extraction": {
                "model": "medium", 
                "temperature": 0.1,
                "max_tokens": 1000,
                "top_k": 10,
                "repeat_penalty": 1.05
            },
            "analysis": {
                "model": "heavy",
                "temperature": 0.3,
                "max_tokens": 3000,
                "top_p": 0.95
            },
            "creative": {
                "model": "medium",
                "temperature": 0.8,
                "max_tokens": 2000,
                "top_p": 0.95,
                "repeat_penalty": 1.2
            }
        }
        
        config = params.get(task_type, params["general"])
        config["messages"] = [{"role": "user", "content": prompt}]
        
        response = requests.post(f"{self.base_url}/chat/completions", json=config)
        return response.json()
```

### 4. Structured Output with Grammar

```python
# Entity extraction with strict JSON grammar
entity_grammar = '''
root ::= object
object ::= "{" ws "\\"entities\\"" ws ":" ws array ws "}"
array ::= "[" ws "]" | "[" ws entity ("," ws entity)* ws "]"
entity ::= "{" ws "\\"name\\"" ws ":" ws string ws "," ws "\\"type\\"" ws ":" ws entity-type ws "}"
entity-type ::= "\\"person\\"" | "\\"organization\\"" | "\\"location\\""
string ::= "\\"" ([^"\\\\] | "\\\\" .)* "\\""
ws ::= [ \\t\\n]*
'''

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "medium",
        "messages": [{"role": "user", "content": "Extract entities from: Apple Inc. is located in Cupertino."}],
        "temperature": 0.1,
        "max_tokens": 200,
        "grammar": entity_grammar
    }
)
```

### 5. Batch Processing with Different Parameters

```python
import asyncio
import aiohttp

async def batch_process_with_params(tasks):
    """Process tasks with task-specific parameters."""
    
    async with aiohttp.ClientSession() as session:
        async def process_one(task):
            # Each task can have completely different parameters
            payload = {
                "model": task.get("model", "medium"),
                "messages": [{"role": "user", "content": task["prompt"]}],
                "temperature": task.get("temperature", 0.3),
                "max_tokens": task.get("max_tokens", 1000),
                "top_p": task.get("top_p", 0.95),
                "seed": task.get("seed")  # Optional seed for reproducibility
            }
            
            # Add custom parameters if specified
            if "repeat_penalty" in task:
                payload["repeat_penalty"] = task["repeat_penalty"]
            if "grammar" in task:
                payload["grammar"] = task["grammar"]
            
            async with session.post(
                "http://localhost:8000/v1/chat/completions",
                json=payload
            ) as response:
                return await response.json()
        
        results = await asyncio.gather(*[process_one(task) for task in tasks])
        return results

# Example usage
tasks = [
    {
        "prompt": "Classify this: breach of contract claim",
        "model": "light",
        "temperature": 0.0,
        "max_tokens": 10
    },
    {
        "prompt": "Extract entities from: Microsoft sued Google in California",
        "model": "medium",
        "temperature": 0.1,
        "max_tokens": 500,
        "seed": 42  # Reproducible
    },
    {
        "prompt": "Write a detailed analysis of patent law",
        "model": "heavy",
        "temperature": 0.5,
        "max_tokens": 4000,
        "repeat_penalty": 1.1
    }
]

results = asyncio.run(batch_process_with_params(tasks))
```

## Parameter Guidelines by Use Case

### Legal Entity Extraction
```python
{
    "temperature": 0.1,      # Low randomness
    "max_tokens": 1000,      # Enough for multiple entities
    "top_k": 10,            # Focused selection
    "repeat_penalty": 1.05,  # Slight penalty to avoid duplicates
    "seed": 42              # Reproducible results
}
```

### Query Classification
```python
{
    "temperature": 0.0,      # Deterministic
    "max_tokens": 20,        # Just the classification
    "top_k": 1              # Greedy decoding
}
```

### Legal Analysis
```python
{
    "temperature": 0.3,      # Some creativity
    "max_tokens": 3000,      # Detailed response
    "top_p": 0.95,          # Balanced sampling
    "repeat_penalty": 1.1    # Avoid repetitive arguments
}
```

### Document Summarization
```python
{
    "temperature": 0.2,      # Mostly factual
    "max_tokens": 500,       # Concise summary
    "top_p": 0.9,           # Slightly focused
    "stop": ["\n\n", "###"] # Stop at section breaks
}
```

## Advanced Features

### 1. Temperature Scheduling

```python
def temperature_schedule(complexity_score):
    """Adjust temperature based on query complexity."""
    if complexity_score < 0.3:
        return 0.1  # Simple queries need deterministic answers
    elif complexity_score < 0.7:
        return 0.3  # Moderate complexity
    else:
        return 0.5  # Complex queries benefit from exploration
```

### 2. Adaptive Token Limits

```python
def adaptive_max_tokens(task_type, input_length):
    """Dynamically set max_tokens based on task and input."""
    base_tokens = {
        "classification": 20,
        "extraction": 500,
        "summary": 300,
        "analysis": 2000
    }
    
    # Scale based on input length
    scale_factor = min(input_length / 1000, 3.0)
    return int(base_tokens.get(task_type, 1000) * scale_factor)
```

### 3. Multi-Model Ensemble

```python
async def ensemble_generate(prompt, models=["light", "medium", "heavy"]):
    """Get responses from multiple models with different parameters."""
    tasks = []
    
    for model in models:
        # Different parameters for each model
        if model == "light":
            params = {"temperature": 0.1, "max_tokens": 100}
        elif model == "medium":
            params = {"temperature": 0.3, "max_tokens": 1000}
        else:
            params = {"temperature": 0.5, "max_tokens": 2000}
        
        tasks.append({
            "model": model,
            "prompt": prompt,
            **params
        })
    
    results = await batch_process_with_params(tasks)
    return results
```

## Performance Considerations

1. **Model Loading**: Models are loaded once at startup, parameter changes don't reload models
2. **Caching**: Responses are cached based on prompt + parameters, so identical requests are fast
3. **Concurrency**: Each model tier can handle multiple requests with different parameters simultaneously
4. **Memory**: Parameters like `max_tokens` and `n_ctx` affect memory usage per request

## Best Practices

1. **Start Conservative**: Begin with low temperature and adjust up if needed
2. **Use Seeds**: For reproducible results in testing/evaluation
3. **Set Stop Sequences**: Prevent runaway generation with appropriate stops
4. **Monitor Token Usage**: Track actual vs requested tokens for cost optimization
5. **Profile Parameters**: Test different parameter combinations for your use case

The key advantage is that **the same loaded model can serve requests with completely different parameters**, making the service highly flexible and efficient for multi-tenant scenarios.