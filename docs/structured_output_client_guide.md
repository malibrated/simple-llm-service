# Structured Output Client Guide

This guide explains how to use the LLM Service's structured output feature from client applications.

## Quick Start

The LLM Service supports structured output through the `response_format` parameter in the chat completions endpoint, following OpenAI's API conventions.

### Basic JSON Object Request

```python
import httpx
import json

# Basic request for JSON output
request = {
    "model": "light",  # or "medium", "heavy", etc.
    "messages": [
        {
            "role": "user",
            "content": "Generate a product with name, price, and description"
        }
    ],
    "response_format": {
        "type": "json_object"
    },
    "temperature": 0.1,  # Lower temperature for more consistent structure
    "max_tokens": 200
}

# Send request
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/v1/chat/completions",
        json=request
    )
    
    result = response.json()
    content = result["choices"][0]["message"]["content"]
    
    # Content will be clean JSON, no markdown
    data = json.loads(content)
    print(data)
    # Output: {"name": "Wireless Mouse", "price": 29.99, "description": "..."}
```

## Supported Format

Currently, the service supports `json_object` format:

```json
{
    "response_format": {
        "type": "json_object"
    }
}
```

This ensures the model generates valid JSON without any markdown formatting or additional text.

## Client Examples

### Python Client

```python
import httpx
import json
from typing import Dict, Any, Optional

class LLMServiceClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    async def generate_json(
        self, 
        prompt: str, 
        model: str = "light",
        temperature: float = 0.1,
        max_tokens: int = 200
    ) -> Dict[str, Any]:
        """Generate structured JSON output."""
        
        request = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=request,
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"Error: {response.status_code} - {response.text}")
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Parse the clean JSON response
            return json.loads(content)

# Usage
async def main():
    client = LLMServiceClient()
    
    # Example 1: Generate product data
    product = await client.generate_json(
        "Generate a product listing for a laptop with all relevant details"
    )
    print(f"Product: {product}")
    
    # Example 2: Extract structured data
    person = await client.generate_json(
        "Create a person profile with name, age, occupation, and hobbies"
    )
    print(f"Person: {person}")
```

### JavaScript/TypeScript Client

```typescript
interface StructuredRequest {
    model: string;
    messages: Array<{role: string, content: string}>;
    response_format: {type: string};
    temperature?: number;
    max_tokens?: number;
}

class LLMServiceClient {
    private baseUrl: string;
    
    constructor(baseUrl: string = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async generateJSON(
        prompt: string, 
        model: string = 'light',
        options: {temperature?: number, maxTokens?: number} = {}
    ): Promise<any> {
        const request: StructuredRequest = {
            model: model,
            messages: [
                {role: 'user', content: prompt}
            ],
            response_format: {type: 'json_object'},
            temperature: options.temperature || 0.1,
            max_tokens: options.maxTokens || 200
        };
        
        const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(request)
        });
        
        if (!response.ok) {
            throw new Error(`Error: ${response.status} - ${await response.text()}`);
        }
        
        const result = await response.json();
        const content = result.choices[0].message.content;
        
        // Parse the clean JSON response
        return JSON.parse(content);
    }
}

// Usage
const client = new LLMServiceClient();

// Generate product data
const product = await client.generateJSON(
    'Generate a product listing for a smartphone'
);
console.log('Product:', product);
```

### cURL Example

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "light",
    "messages": [
      {
        "role": "user",
        "content": "Generate a recipe with ingredients and steps"
      }
    ],
    "response_format": {
      "type": "json_object"
    },
    "temperature": 0.1,
    "max_tokens": 300
  }'
```

## Best Practices

### 1. Prompt Engineering

Be specific about the JSON structure you want:

```python
# Good prompt
prompt = """Generate a product with the following fields:
- name (string)
- price (number)
- inStock (boolean)
- categories (array of strings)"""

# Better prompt with example
prompt = """Generate a product JSON object.
Example structure: {"name": "...", "price": 0.0, "inStock": true, "categories": ["..."]}
Generate a product for a wireless keyboard."""
```

### 2. Temperature Settings

Use lower temperature for more consistent structure:
- `0.0 - 0.3`: Very consistent, predictable structure
- `0.3 - 0.7`: Balanced creativity and structure
- `0.7+`: More creative, may vary structure

### 3. Error Handling

Always handle potential JSON parsing errors:

```python
try:
    data = json.loads(content)
except json.JSONDecodeError as e:
    print(f"Failed to parse JSON: {e}")
    print(f"Raw content: {content}")
    # Handle error or retry
```

### 4. Model Selection

- **light**: Fast responses, good for simple JSON structures
- **medium**: Balanced performance, handles complex structures well
- **heavy**: Best quality, ideal for complex nested JSON

## Response Format

The service returns standard OpenAI-compatible responses:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "light",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\"name\": \"Product\", \"price\": 29.99}"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 25,
    "total_tokens": 40
  }
}
```

The `content` field contains the clean JSON string that can be parsed directly.

## Limitations

1. **Current Format Support**: Only `json_object` is supported. Future versions will add:
   - `json_schema`: Full JSON Schema validation
   - `gbnf_grammar`: Custom GBNF grammars
   - `regex`: Regex pattern matching

2. **No Streaming**: Structured output doesn't support streaming responses

3. **Schema Flexibility**: Currently generates free-form JSON objects. Use clear prompts to guide structure.

## Examples of Common Use Cases

### Data Extraction
```python
response = await client.generate_json(
    "Extract key information from this text: 'Apple Inc. was founded by Steve Jobs in 1976 in Cupertino.'"
)
# Output: {"company": "Apple Inc.", "founder": "Steve Jobs", "year": 1976, "location": "Cupertino"}
```

### List Generation
```python
response = await client.generate_json(
    "List 5 programming languages with their key features"
)
# Output: {"languages": [{"name": "Python", "features": ["readable", "versatile"]}, ...]}
```

### Form Data
```python
response = await client.generate_json(
    "Generate a registration form data for a fictional user"
)
# Output: {"username": "john_doe", "email": "john@example.com", "age": 28, ...}
```

## Troubleshooting

### Issue: Getting markdown-wrapped JSON
**Solution**: Ensure you're including `"response_format": {"type": "json_object"}` in your request.

### Issue: Invalid JSON in response
**Solution**: Check that the model supports structured output and lower the temperature.

### Issue: 400 Bad Request
**Solution**: You're using an unsupported format type. Currently only `"json_object"` is supported.

### Issue: Empty or minimal JSON
**Solution**: Make your prompt more specific about what fields and data you want.

## Integration with Popular Libraries

### LangChain
```python
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="light"
)

# Use with response_format
response = llm.invoke(
    "Generate a product",
    response_format={"type": "json_object"}
)
```

### OpenAI SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="light",
    messages=[{"role": "user", "content": "Generate a product"}],
    response_format={"type": "json_object"}
)
```

This completes the client guide for using structured output with the LLM Service.